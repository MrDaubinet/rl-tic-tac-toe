<script>
	// store
  import { history, selectState, addState } from "../stores/History.ts"
  import { game } from "../game.js"
  
  // Function props
  // export let game

  // $:console.log($history)

  function updateState(index) {
    // $history = [...$history.slice(index)]
    selectState(index)
    addState(game.getStates(), game.getTurn())
    // selectState(index)
    // setSate($history.state)
    // addState(game.getStates(), game.getTurn())
  }
</script>


{#each $history as state, index}
  {#if index != ($history.length -1)}
    <div class="{index == 0 ? 'pb-4' : 'py-4'}">
      <div on:click={() => updateState(index)}
        class="flex justify-center text-2xl cursor-pointer hover:underline">
        <h1> Go Back, </h1>
        <div class="ml-2">
          {#if index == 0}
            to start
          {:else}
            to move {index }
          {/if}
        </div>
      </div>
    </div>
  {/if}
{/each}
